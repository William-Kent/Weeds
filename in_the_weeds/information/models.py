from django.db import models

from modelcluster.fields import ParentalKey

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
    InlinePanel,
)
from wagtail.models import Page, Orderable
from wagtail.fields import RichTextField, StreamField

from streams import blocks

class InfoPageCarouselImages(Orderable):
    """Between 1 and 5 images for the home page carousel."""

    page = ParentalKey("InfoPage", related_name="carousel_images")
    carousel_image = models.ForeignKey(
        "wagtailimages.Image",
        null=True,
        blank=False,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    panels = [FieldPanel("carousel_image")]


class InfoPage(Page):

    template = "information/info_page.html"
    max_count = 1

    weed_title = models.CharField(max_length=100, blank=False, null=True)
    weed_subtitle = RichTextField(features=["bold", "italic"])
    weed_info = RichTextField()
    weed_image = models.ForeignKey(
        "wagtailimages.Image",
        null=True,
        blank=False,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    contentCard = StreamField(
        [
            ("cards", blocks.CardBlock()),
        ],
        null=True,
        use_json_field=True,
        blank=True
    )

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("weed_title"),
                FieldPanel("weed_subtitle"),
                FieldPanel("weed_image"),

            ],
            heading="Banner Options",
        ),
        MultiFieldPanel(
            [InlinePanel("carousel_images", max_num=5, min_num=1, label="Image")],
            heading="Carousel Images",
        ),
        MultiFieldPanel(
        [
        FieldPanel("contentCard"),
        ]
        ),

    ]
    

    class Meta:

        verbose_name = "Information Page"
        verbose_name_plural = "Information Pages"
